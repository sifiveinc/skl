// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "pack/pack_e32rc_e32rcprc.h"
#include "pack/skl_test_pack.h"
#include "skl-test-driver.h"
#include "skl.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_xsfmmbase)
#error This file requires the Xsfmmbase extension
#endif

/**
 * @brief Test cases for pack with Xsfmmbase extension.
 *
 * This test uses the pack_e32rc_e32rcprc harness.
 */

#define TEST                                                                   \
  PACK_E32RC_E32RCPRC_DEFAULTS, .steps = {                                     \
                                    .init = init,                              \
                                    .warmup = NULL,                            \
                                    .execute = execute,                        \
                                    .verify = pack_e32rc_e32rcprc_verify,      \
                                    .report = pack_e32rc_e32rcprc_test_report, \
                                    .cleanup = pack_e32rc_e32rcprc_cleanup,    \
  }

#define BENCH                                                                  \
  PACK_E32RC_E32RCPRC_DEFAULTS,                                                \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = pack_e32rc_e32rcprc_benchmark_report,                      \
          .cleanup = pack_e32rc_e32rcprc_cleanup,                              \
  }

static void init(skl_test_t *t);
static void execute(skl_test_t *t);

// clang-format off
pack_e32rc_e32rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  // Benchmark tests
  {BENCH, .m = 256, .n = 256, .rs = 256, .cs = 1, .rs1 = 16384, .cs1 = 64},
#endif

#ifdef SKL_ENABLE_TESTS
  // Verification tests

  {TEST, .m =  64, .n =  64},
  {TEST, .m =  64, .n = 128},
  {TEST, .m =  64, .n = 192},
  {TEST, .m =  64, .n = 256},
  {TEST, .m = 128, .n =  64},
  {TEST, .m = 128, .n = 128},
  {TEST, .m = 128, .n = 192},
  {TEST, .m = 128, .n = 256},
  {TEST, .m = 192, .n =  64},
  {TEST, .m = 192, .n = 128},
  {TEST, .m = 192, .n = 192},
  {TEST, .m = 192, .n = 256},

  {TEST, .m =  63, .n =  63},
  {TEST, .m =  63, .n = 127},
  {TEST, .m =  63, .n = 191},
  {TEST, .m =  63, .n = 255},
  {TEST, .m = 127, .n =  63},
  {TEST, .m = 127, .n = 127},
  {TEST, .m = 127, .n = 191},
  {TEST, .m = 127, .n = 255},
  {TEST, .m = 191, .n =  63},
  {TEST, .m = 191, .n = 127},
  {TEST, .m = 191, .n = 191},
  {TEST, .m = 191, .n = 255},
#endif
};
// clang-format on

static void init(skl_test_t *t) {
  const pack_e32rc_e32rcprc_t *h = (pack_e32rc_e32rcprc_t *)t->harness;

  size_t ete = skl_get_ete_xsfmmbase();
  SKL_TEST_REQUIRE(t, init_status, h->cs == 1);
  SKL_TEST_REQUIRE(t, init_status, h->m0 == ete);
  SKL_TEST_REQUIRE(t, init_status, h->n0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rs0 == 1); // Note: column-major

  pack_e32rc_e32rcprc_init(t);
}

static skl_test_suite_t suite = {.name = "skl_pack_e32_e32ptex1c_xsfmmbase",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(pack_e32rc_e32rcprc_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  const pack_e32rc_e32rcprc_t *h = (pack_e32rc_e32rcprc_t *)t->harness;

  skl_pack_e32_e32ptex1c_xsfmmbase(h->m, h->n, h->src.data, h->rs, h->dst.data,
                                   h->rs1, h->pad);
}

int main(void) {
  size_t ete = skl_get_ete_xsfmmbase();
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].rs = tests[i].rs ? tests[i].rs : tests[i].n;
    tests[i].cs = 1;

    tests[i].m0 = ete;
    tests[i].n0 = 1;

    tests[i].rs0 = 1;
    tests[i].cs0 = tests[i].m0;
    tests[i].cs1 = tests[i].cs1 ? tests[i].cs1 : tests[i].m0 * tests[i].n0;
    tests[i].rs1 = tests[i].rs1 ? tests[i].rs1 : tests[i].n * tests[i].cs1;
  }
  return skl_test_driver_run_suite(&suite);
}
