// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gemm/gemm_f32rcprc_f32rcprc_f32rcprc.h"
#include "gemm/skl_test_gemm.h"
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
 *  - The block dimensions are M0 = TE, N0 = TE, and K0 = 1
 *  - Matrix A_pack is block-row-major with column-major blocks (rsa0 == 1, csa1
 *    == m0 * k0)
 *  - Matrix B_pack is block-column-major with row-major blocks (csb0 == 1, rsb1
 *    == k0 * n0)
 *  - Matrix C_pack has row-major blocks (csc0 == 1)
 *  - Alpha must be 1.0
 *  - Beta must be 0.0 or 1.0
 *
 * The kernel computes C_pack = A_pack * B_pack (beta = 0) or C_pack += A_pack *
 * B_pack (beta = 1).
 */

#define TEST                                                                   \
  GEMM_F32RCPRC_F32RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_f32rcprc_f32rcprc_f32rcprc_verify,                    \
          .report = gemm_f32rcprc_f32rcprc_f32rcprc_test_report,               \
          .cleanup = gemm_f32rcprc_f32rcprc_f32rcprc_cleanup,                  \
  }

#define BENCH                                                                  \
  GEMM_F32RCPRC_F32RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_f32rcprc_f32rcprc_f32rcprc_benchmark_report,          \
          .cleanup = gemm_f32rcprc_f32rcprc_f32rcprc_cleanup,                  \
  }

static void init(skl_test_t *t);
static void execute(skl_test_t *t);

// clang-format off
gemm_f32rcprc_f32rcprc_f32rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH, .m1 = 2, .n1 = 2, .k1 = 2048, .alpha = 1.f, .beta = 0.f},
    {BENCH, .m1 = 2, .n1 = 2, .k1 = 2048, .alpha = 1.f, .beta = 1.f},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for Xsfmm A1B01
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

    {TEST, .m1 = 7, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 1.f},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {
    .name = "skl_gemm_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f",
    .num_tests = sizeof(tests) / sizeof(tests[0]),
    .test_size = sizeof(gemm_f32rcprc_f32rcprc_f32rcprc_t),
    .tests = tests};

static void init(skl_test_t *t) {
  const gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

  size_t ete = skl_get_ete_xsfmmbase();
  SKL_TEST_REQUIRE(t, init_status, h->m0 == ete);
  SKL_TEST_REQUIRE(t, init_status, h->n0 == ete);
  SKL_TEST_REQUIRE(t, init_status, h->k0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rsa0 == 1); // Note: column-major
  SKL_TEST_REQUIRE(t, init_status, h->csa1 == h->m0 * h->k0);
  SKL_TEST_REQUIRE(t, init_status, h->csb0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rsb1 == h->k0 * h->n0);
  SKL_TEST_REQUIRE(t, init_status, h->rsc0 == h->n0);
  SKL_TEST_REQUIRE(t, init_status, h->csc0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->alpha == 1.f);
  SKL_TEST_REQUIRE(t, init_status, h->beta == 0.f || h->beta == 1.f);

  gemm_f32rcprc_f32rcprc_f32rcprc_init(t);
}

static void execute(skl_test_t *t) {
  const gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

  // Call the kernel with the appropriate parameters
  // The kernel signature is: (m1, n1, k, a_pack, rsa1, b_pack, csb1, c_pack,
  // rsc1, csc1, accum) where accum = (beta != 0)
  skl_gemm_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f(
      h->m1, h->n1, h->k1 * h->k0, h->a_pack.data, h->rsa1, h->b_pack.data,
      h->csb1, h->c_pack.data, h->rsc1, h->csc1, h->beta != 0.f);
}

int main(void) {
  // Set default strides: A is block-row-major with column-major blocks, B is
  // block-column-major with row-major blocks, C has row-major blocks
  size_t ete = skl_get_ete_xsfmmbase();
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].m0 = ete;
    tests[i].n0 = ete;
    tests[i].k0 = 1;

    tests[i].rsa0 = 1;
    tests[i].csa0 = tests[i].m0;
    tests[i].csa1 = tests[i].m0 * tests[i].k0;
    tests[i].rsa1 = tests[i].rsa1 ? tests[i].rsa1 : tests[i].k1 * tests[i].csa1;

    tests[i].rsb0 = tests[i].n0;
    tests[i].csb0 = 1;
    tests[i].rsb1 = tests[i].k0 * tests[i].n0;
    tests[i].csb1 = tests[i].csb1 ? tests[i].csb1 : tests[i].k1 * tests[i].rsb1;

    tests[i].rsc0 = tests[i].n0;
    tests[i].csc0 = 1;
    tests[i].csc1 = tests[i].csc1 ? tests[i].csc1 : tests[i].m0 * tests[i].n0;
    tests[i].rsc1 = tests[i].rsc1 ? tests[i].rsc1 : tests[i].n1 * tests[i].csc1;
  }

  return skl_test_driver_run_suite(&suite);
}
