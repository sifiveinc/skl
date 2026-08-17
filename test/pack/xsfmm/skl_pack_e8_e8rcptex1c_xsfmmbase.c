// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "pack/pack_e8rc_e8rcprc.h"
#include "pack/skl_test_pack.h"
#include "skl-test-driver.h"
#include "skl.h"

#include <stddef.h>

#if !defined(__riscv_xsfmmbase)
#error This file requires the Xsfmmbase extension
#endif

/**
 * @brief Test cases for pack with Xsfmmbase extension.
 *
 * This test uses the pack_e8rc_e8rcprc harness.
 */

#define TEST                                                                   \
  PACK_E8RC_E8RCPRC_DEFAULTS, .steps = {                                       \
                                  .init = init,                                \
                                  .warmup = NULL,                              \
                                  .execute = execute,                          \
                                  .verify = pack_e8rc_e8rcprc_verify,          \
                                  .report = pack_e8rc_e8rcprc_test_report,     \
                                  .cleanup = pack_e8rc_e8rcprc_cleanup,        \
  }

#define BENCH                                                                  \
  PACK_E8RC_E8RCPRC_DEFAULTS,                                                  \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = pack_e8rc_e8rcprc_benchmark_report,                        \
          .cleanup = pack_e8rc_e8rcprc_cleanup,                                \
  }

static void init(skl_test_t *t);
static void execute(skl_test_t *t);

// clang-format off
pack_e8rc_e8rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  // Benchmark tests
  {BENCH, .m = (size_t)4 * SKL_XSFMM_TE, .n = (size_t)4 * SKL_XSFMM_TE},
#endif

#ifdef SKL_ENABLE_TESTS
  // Verification tests
  {TEST, .m = (size_t)1 * SKL_XSFMM_TE, .n = (size_t)1 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)1 * SKL_XSFMM_TE, .n = (size_t)2 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)1 * SKL_XSFMM_TE, .n = (size_t)3 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)1 * SKL_XSFMM_TE, .n = (size_t)4 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE, .n = (size_t)1 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE, .n = (size_t)2 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE, .n = (size_t)3 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE, .n = (size_t)4 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE, .n = (size_t)1 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE, .n = (size_t)2 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE, .n = (size_t)3 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE, .n = (size_t)4 * SKL_XSFMM_TE},

  {TEST, .m = (size_t)1 * SKL_XSFMM_TE - 1, .n = (size_t)1 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)1 * SKL_XSFMM_TE - 1, .n = (size_t)2 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)1 * SKL_XSFMM_TE - 1, .n = (size_t)3 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)1 * SKL_XSFMM_TE - 1, .n = (size_t)4 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE - 1, .n = (size_t)1 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE - 1, .n = (size_t)2 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE - 1, .n = (size_t)3 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)2 * SKL_XSFMM_TE - 1, .n = (size_t)4 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE - 1, .n = (size_t)1 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE - 1, .n = (size_t)2 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE - 1, .n = (size_t)3 * SKL_XSFMM_TE - 1},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE - 1, .n = (size_t)4 * SKL_XSFMM_TE - 1},

  {TEST, .m = (size_t)3 * SKL_XSFMM_TE, .n = (size_t)4 * SKL_XSFMM_TE,
   .cs1 = SKL_XSFMM_TE + 1, .rs1 = (size_t)4 * SKL_XSFMM_TE * (SKL_XSFMM_TE + 1)},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE, .n = (size_t)4 * SKL_XSFMM_TE,
   .rs1 = SKL_XSFMM_TE, .cs1 = (size_t)3 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE, .n = (size_t)4 * SKL_XSFMM_TE,
   .rs1 = SKL_XSFMM_TE + 1, .cs1 = (size_t)3 * (SKL_XSFMM_TE + 1)},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE - 1, .n = (size_t)4 * SKL_XSFMM_TE - 1,
   .cs1 = SKL_XSFMM_TE + 1, .rs1 = (size_t)4 * SKL_XSFMM_TE * (SKL_XSFMM_TE + 1)},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE - 1, .n = (size_t)4 * SKL_XSFMM_TE - 1,
   .rs1 = SKL_XSFMM_TE, .cs1 = (size_t)3 * SKL_XSFMM_TE},
  {TEST, .m = (size_t)3 * SKL_XSFMM_TE - 1, .n = (size_t)4 * SKL_XSFMM_TE - 1,
   .rs1 = SKL_XSFMM_TE + 1, .cs1 = (size_t)3 * (SKL_XSFMM_TE + 1)},
#endif
};
// clang-format on

static void init(skl_test_t *t) {
  const pack_e8rc_e8rcprc_t *h = (pack_e8rc_e8rcprc_t *)t->harness;

  SKL_TEST_REQUIRE(t, init_status, h->cs == 1);
  SKL_TEST_REQUIRE(t, init_status, h->m0 == SKL_XSFMM_TE);
  SKL_TEST_REQUIRE(t, init_status, h->n0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rs0 == 1); // Note: column-major

  pack_e8rc_e8rcprc_init(t);
}

static skl_test_suite_t suite = {.name = "skl_pack_e8_e8rcptex1c_xsfmmbase",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(pack_e8rc_e8rcprc_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  const pack_e8rc_e8rcprc_t *h = (pack_e8rc_e8rcprc_t *)t->harness;

  skl_pack_e8_e8rcptex1c_xsfmmbase(h->m, h->n, h->src.data, h->rs, h->dst.data,
                                   h->rs1, h->cs1, h->pad);
}

int main(void) {
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].rs = tests[i].rs ? tests[i].rs : tests[i].n;
    tests[i].cs = 1;

    tests[i].m0 = SKL_XSFMM_TE;
    tests[i].n0 = 1;

    tests[i].rs0 = 1;
    tests[i].cs0 = tests[i].m0;
    tests[i].cs1 = tests[i].cs1 ? tests[i].cs1 : tests[i].m0 * tests[i].n0;
    tests[i].rs1 = tests[i].rs1 ? tests[i].rs1 : tests[i].n * tests[i].cs1;
  }

  return skl_test_driver_run_suite(&suite);
}
