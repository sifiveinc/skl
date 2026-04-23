// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

/**
 * @brief Test cases for Exponential with Xsfvfexpa extension.
 *
 * This test uses the unary_f32 harness.
 */

#include "elementwise/unary_f32.h"
#include "skl-ref.h" // NOLINT(misc-include-cleaner)
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-104.f)
#define MAX (+89.f)

#define EXP_TESTS(FUN, MIN, MAX, ULP)                                          \
  FUNCTION_TESTS(FUN, skl_exp_f32_ref, (MIN), (MAX), (ULP))
#define EXP_BENCHMARKS FUNCTION_BENCHMARKS

unary_f32_t tests[] = {
#if defined(SKL_ENABLE_PROFILING)
    EXP_BENCHMARKS(skl_exp_1u_f32_xsfvfexpa, MIN, MAX),
    EXP_BENCHMARKS(skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa, -0x1.5d639ep6f,
                   MAX),
#endif
#if defined(SKL_ENABLE_VERIFICATION)
    EXP_TESTS(skl_exp_1u_f32_xsfvfexpa, MIN, MAX, 1.0f),
    EXP_TESTS(skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa, -0x1.5d639ep6f, MAX,
              1.0002f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_f32_xsfvfexpa",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
