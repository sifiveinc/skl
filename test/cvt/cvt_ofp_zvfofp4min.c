// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cvt_ofp.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zvfofp4min)
#error This file requires the Zvfofp4min  extension
#endif

/**
 * @brief Test cases for Conversion from OFP8 to BF16 and F32 to OFP8.
 */

#define TEST_F4E2M1_F8E4M3                                                     \
  .in_type = F4E2M1, .out_type = F8E4M3,                                       \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_f4e2m1_f8e4m3,                                        \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_F4E2M1_F8E4M3                                                    \
  .in_type = F4E2M1, .out_type = F8E4M3,                                       \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_f4e2m1_f8e4m3,                                         \
      .execute = execute_f4e2m1_f8e4m3,                                        \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

static int execute_f4e2m1_f8e4m3(skl_test_t *t);

// clang-format off
cvt_ofp_t tests[] = {
    // Benchmark tests
    {BENCH_F4E2M1_F8E4M3,   .len = 1024},

    // Verification tests
    {TEST_F4E2M1_F8E4M3,    .len = 1024}
};
// clang-format on

static skl_test_suite_t suite = {.name = "cvt_ofp_zvfofp4min",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(cvt_ofp_t),
                                 .tests = tests};

static int execute_f4e2m1_f8e4m3(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_f4e2m1_f8e4m3_zvfofp4min((uint8_t *)h->out, (uint8_t *)h->in, h->len);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

int main(void) { return skl_test_driver_run_suite(&suite); }
