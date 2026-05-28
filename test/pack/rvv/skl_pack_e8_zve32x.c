// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "pack/pack_e8rc_e8rcprc.h"
#include "skl-test-driver.h"
#include "skl.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zve32x)
#error This file requires the Zve32x extension
#endif

/**
 * @brief Test cases for pack with Zve32x extension.
 *
 * This test uses the pack_e8 harness.
 */

#define TEST                                                                   \
  PACK_E8_DEFAULTS, .steps = {                                                 \
                        .init = pack_e8_init,                                  \
                        .warmup = NULL,                                        \
                        .execute = execute,                                    \
                        .verify = pack_e8_verify,                              \
                        .report = pack_e8_param_report,                        \
                        .cleanup = pack_e8_cleanup,                            \
  }

#define BENCH                                                                  \
  PACK_E8_DEFAULTS, .steps = {                                                 \
                        .init = pack_e8_init,                                  \
                        .warmup = execute,                                     \
                        .execute = execute,                                    \
                        .verify = NULL,                                        \
                        .report = pack_e8_bench_report,                        \
                        .cleanup = pack_e8_cleanup,                            \
  }

static void execute(skl_test_t *t);

// clang-format off
pack_e8_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  // Benchmark tests
  {BENCH, .m = 128, .n = 128, .rs = 128, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 1024, .cs1 = 64},
#endif

#ifdef SKL_ENABLE_TESTS
  // Verification tests

  /* Small matrices (one-block case)*/
  {TEST, .m = 2, .n = 3, .rs = 3, .cs = 1, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 4, .rs1 = 16, .cs1 = 16},
  {TEST, .m = 2, .n = 3, .rs = 1, .cs = 2, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 4, .rs1 = 16, .cs1 = 16},
  {TEST, .m = 2, .n = 3, .rs = 3, .cs = 1, .m0 = 4, .n0 = 4,
   .rs0 = 4, .cs0 = 1, .rs1 = 16, .cs1 = 16},
  {TEST, .m = 2, .n = 3, .rs = 1, .cs = 2, .m0 = 4, .n0 = 4,
   .rs0 = 4, .cs0 = 1, .rs1 = 16, .cs1 = 16},

  /* Non-multiple of block size */
  {TEST, .m = 5, .n = 5, .rs = 5, .cs = 1, .m0 = 4, .n0 = 8,
   .rs0 = 1, .cs0 = 4, .rs1 = 64, .cs1 = 32},
  {TEST, .m = 10, .n = 10, .rs = 10, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 128, .cs1 = 64},
  {TEST, .m = 7, .n = 9, .rs = 9, .cs = 1, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 4, .rs1 = 48, .cs1 = 16},
  {TEST, .m = 9, .n = 8, .rs = 8, .cs = 1, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 4, .rs1 = 32, .cs1 = 16},
  {TEST, .m = 20, .n = 16, .rs = 16, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 128, .cs1 = 64},

  {TEST, .m = 5, .n = 5, .rs = 1, .cs = 5, .m0 = 4, .n0 = 8,
   .rs0 = 1, .cs0 = 4, .rs1 = 64, .cs1 = 32},
  {TEST, .m = 10, .n = 10, .rs = 1, .cs = 10, .m0 = 8, .n0 = 8,
   .rs0 = 8, .cs0 = 1, .rs1 = 128, .cs1 = 64},
  {TEST, .m = 7, .n = 9, .rs = 1, .cs = 7, .m0 = 4, .n0 = 4,
   .rs0 = 4, .cs0 = 1, .rs1 = 48, .cs1 = 16},
  {TEST, .m = 9, .n = 8, .rs = 1, .cs = 9, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 4, .rs1 = 32, .cs1 = 16},
  {TEST, .m = 20, .n = 16, .rs = 1, .cs = 20, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 128, .cs1 = 64},


  /* Multiple of block size  */
  {TEST, .m = 8, .n = 8, .rs = 16, .cs = 1, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 8, .rs1 = 64, .cs1 = 32},
  {TEST, .m = 12, .n = 12, .rs = 12, .cs = 1, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 4, .rs1 = 48, .cs1 = 16},
  {TEST, .m = 16, .n = 16, .rs = 16, .cs = 1, .m0 = 4, .n0 = 4,
   .rs0 = 1, .cs0 = 4, .rs1 = 64, .cs1 = 16},
  {TEST, .m = 16, .n = 16, .rs = 16, .cs = 1, .m0 = 4, .n0 = 8,
   .rs0 = 1, .cs0 = 4, .rs1 = 64, .cs1 = 32},
  {TEST, .m = 16, .n = 16, .rs = 16, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 128, .cs1 = 64},
  {TEST, .m = 16, .n = 16, .rs = 16, .cs = 1, .m0 = 16, .n0 = 16,
   .rs0 = 1, .cs0 = 16, .rs1 = 256, .cs1 = 256},
  {TEST, .m = 32, .n = 8, .rs = 8, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 64, .cs1 = 64},
  {TEST, .m = 8, .n = 32, .rs = 32, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 256, .cs1 = 64},

  /* Specialization 1: (cs0 * n0 == cs1) && rs0 == 1 */
  {TEST, .m = 16, .n = 12, .rs = 12, .cs = 1, .m0 = 4, .n0 = 3,
   .rs0 = 1, .cs0 = 4, .rs1 = 48, .cs1 = 12},
  {TEST, .m = 10, .n = 8, .rs = 8, .cs = 1, .m0 = 4, .n0 = 3,
   .rs0 = 1, .cs0 = 4, .rs1 = 36, .cs1 = 12},
  {TEST, .m = 10, .n = 18, .rs = 18, .cs = 1, .m0 = 4, .n0 = 1,
   .rs0 = 1, .cs0 = 4, .rs1 = 72, .cs1 = 4},

  /* Specialization 2: (rs0 * m0 == rs1) && cs0 == 1 */
  {TEST, .m = 12, .n = 16, .rs = 16, .cs = 1, .m0 = 4, .n0 = 8,
   .rs0 = 8, .cs0 = 1, .rs1 = 32, .cs1 = 96},
  {TEST, .m = 16, .n = 12, .rs = 12, .cs = 1, .m0 = 2, .n0 = 5,
   .rs0 = 5, .cs0 = 1, .rs1 = 10, .cs1 = 80},
  {TEST, .m = 10, .n = 8, .rs = 8, .cs = 1, .m0 = 4, .n0 = 8,
   .rs0 = 8, .cs0 = 1, .rs1 = 32, .cs1 = 32},

  /* Specialization 3: rs0 == 1 && n0 == 1 && m0 == rs1 - Column panel transpose */
  {TEST, .m = 12, .n = 8, .rs = 8, .cs = 1, .m0 = 4, .n0 = 1,
   .rs0 = 1, .cs0 = 4, .rs1 = 4, .cs1 = 12},
  {TEST, .m = 20, .n = 16, .rs = 16, .cs = 1, .m0 = 8, .n0 = 1,
   .rs0 = 1, .cs0 = 8, .rs1 = 8, .cs1 = 24},

  /* Specialization 4: cs0 == 1 && m0 == 1 && n0 == cs1 - Row panel copy */
  {TEST, .m = 8, .n = 12, .rs = 12, .cs = 1, .m0 = 1, .n0 = 4,
   .rs0 = 4, .cs0 = 1, .rs1 = 12, .cs1 = 4},
  {TEST, .m = 12, .n = 10, .rs = 10, .cs = 1, .m0 = 1, .n0 = 4,
   .rs0 = 4, .cs0 = 1, .rs1 = 12, .cs1 = 4},

  /* Larger matrices */
  {TEST, .m = 64, .n = 64, .rs = 64, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 512, .cs1 = 64},
  {TEST, .m = 128, .n = 128, .rs = 128, .cs = 1, .m0 = 8, .n0 = 8,
   .rs0 = 1, .cs0 = 8, .rs1 = 1024, .cs1 = 64},

  {TEST, .m = 300, .n = 200, .rs = 200, .cs = 1, .m0 = 64, .n0 = 16,
   .rs0 = 16, .cs0 = 1, .rs1 = 13312, .cs1 = 1024},
  {TEST, .m = 300, .n = 200, .rs = 200, .cs = 1, .m0 = 64, .n0 = 16,
   .rs0 = 16, .cs0 = 1, .rs1 = 1024, .cs1 = 5120},
  {TEST, .m = 300, .n = 200, .rs = 200, .cs = 1, .m0 = 64, .n0 = 16,
   .rs0 = 1, .cs0 = 64, .rs1 = 13312, .cs1 = 1024},
  {TEST, .m = 300, .n = 200, .rs = 200, .cs = 1, .m0 = 64, .n0 = 16,
   .rs0 = 1, .cs0 = 64, .rs1 = 1024, .cs1 = 5120},

  {TEST, .m = 300, .n = 200, .rs = 1, .cs = 300, .m0 = 64, .n0 = 16,
   .rs0 = 16, .cs0 = 1, .rs1 = 13312, .cs1 = 1024},
  {TEST, .m = 300, .n = 200, .rs = 1, .cs = 300, .m0 = 64, .n0 = 16,
   .rs0 = 16, .cs0 = 1, .rs1 = 1024, .cs1 = 5120},
  {TEST, .m = 300, .n = 200, .rs = 1, .cs = 300, .m0 = 64, .n0 = 16,
   .rs0 = 1, .cs0 = 64, .rs1 = 13312, .cs1 = 1024},
  {TEST, .m = 300, .n = 200, .rs = 1, .cs = 300, .m0 = 64, .n0 = 16,
   .rs0 = 1, .cs0 = 64, .rs1 = 1024, .cs1 = 5120},
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_pack_e8_zve32x",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(pack_e8_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  const pack_e8_t *h = (pack_e8_t *)t->harness;

  skl_pack_e8rc_e8rcprc_zve32x(h->m, h->n, h->src.data, h->rs, h->cs, h->m0,
                               h->n0, h->dst.data, h->rs0, h->cs0, h->rs1,
                               h->cs1, h->pad);
}

int main(void) { return skl_test_driver_run_suite(&suite); }
